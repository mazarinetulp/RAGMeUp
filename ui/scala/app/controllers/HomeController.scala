package controllers

import javax.inject._
import play.api._

import java.nio.file.{FileSystems, Files}
import play.api.libs.json._
import play.api.mvc._
import play.api.libs.ws._

import scala.concurrent.{ExecutionContext, Future}
import scala.jdk.CollectionConverters._
import scala.util.Try

@Singleton
class HomeController @Inject()(
    cc: ControllerComponents,
    config: Configuration,
    ws: WSClient
) (implicit ec: ExecutionContext) extends AbstractController(cc) {

  def index() = Action { implicit request: Request[AnyContent] =>
    Ok(views.html.index())
  }

  def add() = Action { implicit request: Request[AnyContent] =>
    val files = Try {
      Files.list(FileSystems.getDefault.getPath(config.get[String]("data_folder"))).iterator().asScala.map(_.getFileName.toString).toSeq
    } getOrElse(Nil)
    Ok(views.html.add(files))
  }

  def search() = Action.async { implicit request: Request[AnyContent] =>
    val json = request.body.asJson.getOrElse(Json.obj()).as[JsObject]
    println(json)
    val query = (json \ "query").as[String]
    val history = (json \ "history").as[String]

    ws
      .url(s"${config.get[String]("server_url")}/chat")
      .post(Json.obj(
        "prompt" -> query,
        "history" -> history
      ))
      .map(response =>
          Ok(response.json)
      )
  }
}
